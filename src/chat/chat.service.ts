import { Injectable, InternalServerErrorException } from '@nestjs/common';
import { HttpService } from '@nestjs/axios';
import { DatabaseService } from '../database/database.service';
import { CulturalProfileService } from '../cultural-profile/cultural-profile.service';
import { ConversationSessionService } from '../conversation-session/conversation-session.service';
import { firstValueFrom } from 'rxjs';
import {
  AIServiceRequest,
  AIServiceResponse,
  ConversationMemory,
} from '../common/interfaces/ai-service.interface';

@Injectable()
export class ChatService {
  constructor(
    private prisma: DatabaseService,
    private httpService: HttpService,
    private culturalProfileService: CulturalProfileService,
    private sessionService: ConversationSessionService,
  ) {}

  async checkAIServiceHealth(): Promise<boolean> {
    try {
      const response = await firstValueFrom(
        this.httpService.get('http://localhost:8000/health', {
          timeout: 5000,
        }),
      );
      return response.data.status === 'healthy';
    } catch (error) {
      console.error('AI service health check failed:', error.message);
      return false;
    }
  }

  async handleMessage(userId: string, message: string, sessionId?: string) {
    try {
      // Optional: Check AI service health before processing
      const isHealthy = await this.checkAIServiceHealth();
      if (!isHealthy) {
        throw new InternalServerErrorException(
          'AI service is currently unavailable. Please try again in a few moments.',
        );
      }
      // Get or create active session
      const session = sessionId
        ? await this.sessionService.getSessionWithChats(sessionId)
        : await this.sessionService.getOrCreateActiveSession(userId);

      if (!session) {
        throw new InternalServerErrorException('Failed to create session');
      }

      // Get user's cultural profile
      const culturalProfile =
        await this.culturalProfileService.getOrCreateProfile(userId);

      // Build conversation memory
      const conversationMemory = await this.buildConversationMemory(
        userId,
        session.id,
      );

      // Count messages in this session to determine conversation stage
      const sessionMessageCount = conversationMemory.recentMessages
        ? conversationMemory.recentMessages.filter(
            (msg) => msg.sender === 'USER',
          ).length
        : 0;

      // Save user message first
      const userMessage = await this.prisma.chat.create({
        data: {
          userId,
          sender: 'USER',
          message,
          sessionId: session.id,
        },
      });

      // Prepare AI service request
      const aiRequest: AIServiceRequest = {
        message,
        userId,
        sessionId: session.id,
        culturalProfile: {
          id: culturalProfile.id,
          ageGroup: culturalProfile.ageGroup,
          location: culturalProfile.location,
          educationLevel: culturalProfile.educationLevel,
          ethnicBackground: culturalProfile.ethnicBackground || undefined,
          religiousBackground: culturalProfile.religiousBackground || undefined,
          languagePreference: culturalProfile.languagePreference || undefined,
          familyStructure: culturalProfile.familyStructure,
          householdSize: culturalProfile.householdSize || undefined,
          hasElders: culturalProfile.hasElders,
          communicationStyle: culturalProfile.communicationStyle || undefined,
          respectLevel: culturalProfile.respectLevel,
          economicStatus: culturalProfile.economicStatus,
          employmentStatus: culturalProfile.employmentStatus || undefined,
        },
        conversationMemory,
      };

      // Determine message type and distress level for dynamic parameters
      const messageType = this.classifyMessage(message);
      const distressLevel = this.detectDistress(message);

      // Adjust AI parameters based on message type and distress
      const temperature = this.getTemperatureForMessageType(messageType);
      const maxTokens = this.getMaxTokensForMessageType(messageType);

      // Call FastAPI AI inference service with stage-aware context
      const fastApiRequest = {
        message,
        context: this.buildContextString(
          conversationMemory,
          message,
          sessionMessageCount,
        ),
        parameters: {
          temperature,
          max_tokens: maxTokens,
        },
      };

      const response = await firstValueFrom(
        this.httpService.post('http://localhost:8000/infer', fastApiRequest, {
          timeout: 30000, // 30 second timeout
          headers: {
            'Content-Type': 'application/json',
          },
        }),
      );

      const aiResponse = response.data;

      // Save bot message with FastAPI response metadata
      const botMessage = await this.prisma.chat.create({
        data: {
          userId,
          sender: 'BOT',
          message: aiResponse.response,
          sessionId: session.id,
          emotionalTone: aiResponse.metadata?.mood_detected || null,
          detectedTopics: aiResponse.suggested_resources || [],
          culturalContext: {
            cultural_elements: aiResponse.cultural_elements_detected || [],
            quality_metrics: aiResponse.quality_metrics || {},
            confidence: aiResponse.metadata?.confidence || 0,
            emotional_intensity: aiResponse.metadata?.emotional_intensity || 5,
          },
          memoryReferences: [], // Could be populated based on AI response
        },
      });

      // Update session with AI insights from analysis
      const sessionUpdates: any = {};

      if (aiResponse.metadata?.mood_detected) {
        sessionUpdates.emotionalState = aiResponse.metadata.mood_detected;
      }

      if (
        aiResponse.suggested_resources &&
        aiResponse.suggested_resources.length > 0
      ) {
        sessionUpdates.primaryTopic = aiResponse.suggested_resources[0]; // Use first suggested resource as primary topic
        sessionUpdates.actionItems = aiResponse.suggested_resources;
      }

      if (Object.keys(sessionUpdates).length > 0) {
        await this.sessionService.updateSession(session.id, sessionUpdates);
      }

      // Update user topics based on suggested resources and cultural elements
      const detectedTopics = [
        ...(aiResponse.suggested_resources || []),
        ...(aiResponse.cultural_elements_detected || []),
      ];

      if (detectedTopics.length > 0) {
        await this.updateUserTopics(userId, detectedTopics);
      }

      // Update progress tracking based on AI analysis
      if (aiResponse.metadata) {
        await this.updateProgressTracking(userId, {
          relationshipStrength: this.calculateRelationshipStrength(aiResponse),
          trustLevel: aiResponse.metadata.confidence || 0.5,
          improvementAreas: this.extractImprovementAreas(aiResponse),
          concernAreas: this.extractConcernAreas(aiResponse),
        });
      }

      return {
        userMessage: {
          id: userMessage.id,
          message: userMessage.message,
          sender: userMessage.sender,
          timestamp: userMessage.timestamp,
        },
        botMessage: {
          id: botMessage.id,
          message: botMessage.message,
          sender: botMessage.sender,
          timestamp: botMessage.timestamp,
          emotionalTone: botMessage.emotionalTone,
          detectedTopics: botMessage.detectedTopics,
          // Include FastAPI analysis results
          analysisResults: {
            mood_detected: aiResponse.metadata?.mood_detected,
            confidence: aiResponse.metadata?.confidence,
            emotional_intensity: aiResponse.metadata?.emotional_intensity,
            suggested_resources: aiResponse.suggested_resources,
            cultural_elements_detected: aiResponse.cultural_elements_detected,
            quality_metrics: aiResponse.quality_metrics,
          },
        },
        session: {
          id: session.id,
          title: session.title || undefined,
        },
      };
    } catch (err) {
      console.error('ChatService error:', err);

      // Handle specific FastAPI service errors
      if (err.response) {
        const status = err.response.status;
        const errorData = err.response.data;

        if (status === 503) {
          throw new InternalServerErrorException(
            'AI service is currently unavailable. The model may be loading or experiencing issues.',
          );
        } else if (status === 504) {
          throw new InternalServerErrorException(
            'AI service request timed out. Please try again with a shorter message.',
          );
        } else if (status >= 400 && status < 500) {
          throw new InternalServerErrorException(
            `AI service error: ${errorData?.message || 'Invalid request'}`,
          );
        } else {
          throw new InternalServerErrorException(
            'AI service is experiencing technical difficulties. Please try again later.',
          );
        }
      } else if (err.code === 'ECONNREFUSED') {
        throw new InternalServerErrorException(
          'AI service is not available. Please ensure the FastAPI service is running on localhost:8000.',
        );
      } else if (err.code === 'ETIMEDOUT') {
        throw new InternalServerErrorException(
          'AI service request timed out. Please try again.',
        );
      } else {
        throw new InternalServerErrorException(
          'Failed to process message due to an unexpected error.',
        );
      }
    }
  }

  private async buildConversationMemory(
    userId: string,
    sessionId: string,
  ): Promise<ConversationMemory> {
    // Get recent messages
    const recentMessages = await this.sessionService.getRecentChats(userId, 10);

    // Get user topics
    const userTopics = await this.sessionService.getUserTopics(userId);

    // Get session context
    const sessionWithChats =
      await this.sessionService.getSessionWithChats(sessionId);

    // Get latest progress tracking
    const progressTracking =
      await this.sessionService.getLatestProgressTracking(userId);

    return {
      recentMessages: recentMessages.map((chat) => ({
        id: chat.id,
        message: chat.message,
        sender: chat.sender,
        timestamp: chat.timestamp,
        emotionalTone: chat.emotionalTone || undefined,
        detectedTopics: chat.detectedTopics,
      })),
      userTopics: userTopics.map((topic) => ({
        topic: topic.topic,
        frequency: topic.frequency,
        severity: topic.severity,
        status: topic.status,
        lastDiscussed: topic.lastDiscussed,
      })),
      sessionContext: {
        primaryTopic: sessionWithChats?.primaryTopic || undefined,
        emotionalState: sessionWithChats?.emotionalState || undefined,
        keyInsights: sessionWithChats?.keyInsights,
        actionItems: sessionWithChats?.actionItems || [],
      },
      progressTracking: {
        relationshipStrength: progressTracking?.relationshipStrength || 0.0,
        trustLevel: progressTracking?.trustLevel || 0.5,
        improvementAreas: progressTracking?.improvementAreas || [],
        concernAreas: progressTracking?.concernAreas || [],
      },
    };
  }

  /**
   * Get temperature setting based on message type
   */
  private getTemperatureForMessageType(
    messageType: 'greeting' | 'follow_up' | 'new_topic' | 'light_chat',
  ): number {
    switch (messageType) {
      case 'greeting':
        return 0.7; // More consistent, friendly responses
      case 'light_chat':
        return 0.75; // Conversational but not too creative
      case 'follow_up':
        return 0.85; // More creative when user is engaged
      case 'new_topic':
        return 0.8; // Default balanced creativity
    }
  }

  /**
   * Get max tokens based on message type
   */
  private getMaxTokensForMessageType(
    messageType: 'greeting' | 'follow_up' | 'new_topic' | 'light_chat',
  ): number {
    switch (messageType) {
      case 'greeting':
        return 100; // Short, friendly greetings
      case 'light_chat':
        return 150; // Brief conversational responses
      case 'follow_up':
        return 250; // More detailed when user wants depth
      case 'new_topic':
        return 200; // Standard response length
    }
  }

  /**
   * Detect if user is in distress and needs therapeutic support
   */
  private detectDistress(
    message: string,
  ): 'none' | 'mild' | 'moderate' | 'severe' {
    const normalizedMsg = message.toLowerCase();

    // Severe distress indicators - immediate therapeutic mode
    const severeIndicators = [
      /\b(suicid|kill myself|end my life|want to die|harm myself)\b/i,
      /\b(can't take|cannot take|can't cope|cannot cope) (it|this|anymore)\b/i,
      /\b(severe|extreme|unbearable) (pain|depression|anxiety)\b/i,
    ];
    if (severeIndicators.some((pattern) => pattern.test(message))) {
      return 'severe';
    }

    // Moderate distress - explicit help seeking
    const moderateIndicators = [
      /\b(need help|struggling|not okay|feeling (depressed|anxious|hopeless|worthless))\b/i,
      /\b(can't sleep|cannot sleep|insomnia|nightmares) .*(weeks|months|every night)\b/i,
      /\b(panic attack|mental breakdown|crisis)\b/i,
      /\b(really (sad|down|low|bad)|very (depressed|anxious|worried))\b/i,
    ];
    if (moderateIndicators.some((pattern) => pattern.test(message))) {
      return 'moderate';
    }

    // Mild distress - expressing difficulty but not crisis
    const mildIndicators = [
      /\b(stressed|worried|concerned|anxious|sad|down|upset|frustrated)\b/i,
      /\b(having a hard time|difficult time|tough time|going through)\b/i,
      /\b(problem|issue|challenge|struggle) with\b/i,
      /\b(feel|feeling) (bad|terrible|awful|horrible)\b/i,
    ];

    // Only count as mild if message is substantive (not just "I'm bored")
    if (
      message.split(' ').length > 5 &&
      mildIndicators.some((pattern) => pattern.test(message))
    ) {
      return 'mild';
    }

    return 'none';
  }

  /**
   * Classify message type to determine context needs
   */
  private classifyMessage(
    message: string,
  ): 'greeting' | 'follow_up' | 'new_topic' | 'light_chat' {
    const normalizedMsg = message.trim().toLowerCase();

    // Simple greetings
    const greetingPatterns = [
      /^(hi|hey|hello|hii+|heyy+|yo|sup|wassup|howdy)[\s!?.]*$/i,
      /^(good\s+(morning|afternoon|evening|night))[\s!?.]*$/i,
    ];
    if (greetingPatterns.some((pattern) => pattern.test(normalizedMsg))) {
      return 'greeting';
    }

    // Light conversational messages (short, casual, no heavy topics)
    const lightChatPatterns = [
      /^(how are you|how're you|hru|how r u)[\s!?.]*$/i,
      /^(what'?s up|whats up|sup)[\s!?.]*$/i,
      /^(yeah|yep|yes|okay|ok|sure|alright|cool|nice|thanks|thank you)[\s!?.]*$/i,
      /^(lol|haha|hehe)[\s!?.]*$/i,
    ];
    if (
      lightChatPatterns.some((pattern) => pattern.test(normalizedMsg)) ||
      message.length < 20
    ) {
      return 'light_chat';
    }

    // Check if message references previous conversation
    const followUpIndicators = [
      /\b(that|this|you said|you mentioned|earlier|before|previously|last time)\b/i,
      /\b(still|also|too|as well|and)\b/i,
      /\b(continue|more about|tell me more)\b/i,
    ];
    if (followUpIndicators.some((pattern) => pattern.test(message))) {
      return 'follow_up';
    }

    // Default: new topic (needs context awareness but not forced recall)
    return 'new_topic';
  }

  /**
   * Build context string dynamically based on message type, distress level, and conversation flow
   */
  private buildContextString(
    conversationMemory: ConversationMemory,
    currentMessage: string,
    sessionMessageCount: number,
  ): string {
    const messageType = this.classifyMessage(currentMessage);
    const distressLevel = this.detectDistress(currentMessage);
    const contextParts: string[] = [];

    // Add conversation stage indicator
    const conversationStage =
      sessionMessageCount <= 8 ? 'early' : 'established';
    contextParts.push(
      `[Conversation stage: ${conversationStage}, Message #${sessionMessageCount + 1}]`,
    );

    // FORCE casual mode for early conversation unless severe distress
    if (
      conversationStage === 'early' &&
      distressLevel !== 'severe' &&
      distressLevel !== 'moderate'
    ) {
      // Early conversation - stay casual even if mild distress detected
      if (
        conversationMemory.recentMessages &&
        conversationMemory.recentMessages.length > 0
      ) {
        const recentContext = conversationMemory.recentMessages
          .slice(-2)
          .map((msg) => `${msg.sender}: ${msg.message}`)
          .join('\n');
        contextParts.push(`Recent exchange:\n${recentContext}`);
      }
      contextParts.push(
        `[INSTRUCTION: This is early conversation. Stay casual and friendly. Build rapport naturally. Do NOT assume user needs therapy unless they explicitly ask for help.]`,
      );
      return contextParts.join('\n\n');
    }

    // OVERRIDE for explicit distress - go therapeutic immediately
    if (distressLevel === 'severe' || distressLevel === 'moderate') {
      // Full therapeutic context
      if (
        conversationMemory.recentMessages &&
        conversationMemory.recentMessages.length > 0
      ) {
        const recentContext = conversationMemory.recentMessages
          .slice(-4)
          .map((msg) => `${msg.sender}: ${msg.message}`)
          .join('\n');
        contextParts.push(`Recent conversation:\n${recentContext}`);
      }

      if (
        conversationMemory.userTopics &&
        conversationMemory.userTopics.length > 0
      ) {
        const relevantTopics = conversationMemory.userTopics.slice(0, 3);
        const topicContext = relevantTopics
          .map((topic) => `${topic.topic} (${topic.severity})`)
          .join(', ');
        contextParts.push(`User's ongoing topics: ${topicContext}`);
      }

      contextParts.push(
        `[INSTRUCTION: User is expressing ${distressLevel} distress. Provide appropriate therapeutic support.]`,
      );
      return contextParts.join('\n\n');
    }

    // Default handling based on message type (for established conversations or mild distress)
    switch (messageType) {
      case 'greeting':
        if (
          conversationMemory.recentMessages &&
          conversationMemory.recentMessages.length > 0
        ) {
          const lastMessage =
            conversationMemory.recentMessages[
              conversationMemory.recentMessages.length - 1
            ];
          if (lastMessage) {
            contextParts.push(
              `Last interaction: ${lastMessage.sender}: "${lastMessage.message.slice(0, 50)}${lastMessage.message.length > 50 ? '...' : ''}"`,
            );
          }
        }
        break;

      case 'light_chat':
        if (
          conversationMemory.recentMessages &&
          conversationMemory.recentMessages.length > 0
        ) {
          const recentContext = conversationMemory.recentMessages
            .slice(-2)
            .map((msg) => `${msg.sender}: ${msg.message}`)
            .join('\n');
          contextParts.push(`Recent exchange:\n${recentContext}`);
        }
        break;

      case 'follow_up':
        if (
          conversationMemory.recentMessages &&
          conversationMemory.recentMessages.length > 0
        ) {
          const recentContext = conversationMemory.recentMessages
            .slice(-4)
            .map((msg) => `${msg.sender}: ${msg.message}`)
            .join('\n');
          contextParts.push(`Recent conversation:\n${recentContext}`);
        }

        if (
          conversationMemory.userTopics &&
          conversationMemory.userTopics.length > 0
        ) {
          const recentTopics = conversationMemory.userTopics
            .filter((topic) => {
              const daysSinceDiscussed =
                (new Date().getTime() -
                  new Date(topic.lastDiscussed).getTime()) /
                (1000 * 60 * 60 * 24);
              return daysSinceDiscussed < 7;
            })
            .slice(0, 2);

          if (recentTopics.length > 0) {
            const topicContext = recentTopics
              .map((topic) => `${topic.topic}`)
              .join(', ');
            contextParts.push(`Recent discussion topics: ${topicContext}`);
          }
        }
        break;

      case 'new_topic':
        if (
          conversationMemory.recentMessages &&
          conversationMemory.recentMessages.length > 0
        ) {
          const recentContext = conversationMemory.recentMessages
            .slice(-3)
            .map((msg) => `${msg.sender}: ${msg.message}`)
            .join('\n');
          contextParts.push(`Recent conversation:\n${recentContext}`);
        }

        // Only add background topics if conversation is established
        if (
          conversationStage === 'established' &&
          conversationMemory.userTopics &&
          conversationMemory.userTopics.length > 0
        ) {
          const activeTopics = conversationMemory.userTopics
            .filter(
              (topic) =>
                topic.status === 'ONGOING' && topic.severity !== 'MILD',
            )
            .slice(0, 2);

          if (activeTopics.length > 0) {
            const topicContext = activeTopics
              .map((topic) => topic.topic)
              .join(', ');
            contextParts.push(
              `Background awareness: User has discussed ${topicContext} (reference only if relevant)`,
            );
          }
        }
        break;
    }

    return contextParts.join('\n\n');
  }

  private calculateRelationshipStrength(aiResponse: any): number {
    // Calculate relationship strength based on response quality and cultural awareness
    const baseStrength = 0.5;
    const qualityBonus = (aiResponse.quality_metrics?.empathy_score || 0) * 0.3;
    const culturalBonus =
      (aiResponse.quality_metrics?.cultural_awareness_score || 0) * 0.2;

    return Math.min(baseStrength + qualityBonus + culturalBonus, 1.0);
  }

  private extractImprovementAreas(aiResponse: any): string[] {
    const improvements: string[] = [];

    // If mood is positive or neutral, consider suggested resources as improvement areas
    if (
      aiResponse.metadata?.mood_detected === 'positive' ||
      aiResponse.metadata?.mood_detected === 'neutral'
    ) {
      improvements.push(...(aiResponse.suggested_resources || []));
    }

    return improvements;
  }

  private extractConcernAreas(aiResponse: any): string[] {
    const concerns: string[] = [];

    // If mood is negative with high intensity, mark suggested resources as concerns
    if (
      aiResponse.metadata?.mood_detected === 'negative' &&
      (aiResponse.metadata?.emotional_intensity || 0) >= 7
    ) {
      concerns.push(...(aiResponse.suggested_resources || []));
    }

    return concerns;
  }

  private async updateUserTopics(userId: string, detectedTopics: string[]) {
    for (const topic of detectedTopics) {
      await this.prisma.userTopic.upsert({
        where: {
          userId_topic: { userId, topic },
        },
        update: {
          frequency: { increment: 1 },
          lastDiscussed: new Date(),
        },
        create: {
          userId,
          topic,
          frequency: 1,
          severity: 'MODERATE',
          status: 'ONGOING',
        },
      });
    }
  }

  private async updateProgressTracking(
    userId: string,
    progressUpdates: {
      relationshipStrength?: number;
      trustLevel?: number;
      improvementAreas?: string[];
      concernAreas?: string[];
    },
  ) {
    await this.prisma.progressTracking.create({
      data: {
        userId,
        relationshipStrength: progressUpdates.relationshipStrength || 0.0,
        trustLevel: progressUpdates.trustLevel || 0.5,
        improvementAreas: progressUpdates.improvementAreas || [],
        concernAreas: progressUpdates.concernAreas || [],
      },
    });
  }

  async getUserSessions(userId: string) {
    const sessions = await this.prisma.conversationSession.findMany({
      where: { userId },
      orderBy: { startTime: 'desc' },
      select: {
        id: true,
        title: true,
        startTime: true,
        endTime: true,
        primaryTopic: true,
        emotionalState: true,
        followUpNeeded: true,
        _count: {
          select: { chats: true },
        },
      },
    });

    return sessions;
  }
}
