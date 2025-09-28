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

      // Call FastAPI AI inference service
      const fastApiRequest = {
        message,
        context: this.buildContextString(conversationMemory),
        parameters: {
          temperature: 0.8,
          max_tokens: 200,
        },
      };

      const response = await firstValueFrom(
        this.httpService.post(
          'http://localhost:8000/infer',
          fastApiRequest,
          {
            timeout: 30000, // 30 second timeout
            headers: {
              'Content-Type': 'application/json',
            },
          },
        ),
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

      if (aiResponse.suggested_resources && aiResponse.suggested_resources.length > 0) {
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

  private buildContextString(conversationMemory: ConversationMemory): string {
    const contextParts: string[] = [];

    // Add recent conversation context
    if (conversationMemory.recentMessages && conversationMemory.recentMessages.length > 0) {
      const recentContext = conversationMemory.recentMessages
        .slice(-3) // Last 3 messages for context
        .map(msg => `${msg.sender}: ${msg.message}`)
        .join('\n');
      contextParts.push(`Recent conversation:\n${recentContext}`);
    }

    // Add ongoing topics
    if (conversationMemory.userTopics && conversationMemory.userTopics.length > 0) {
      const topicContext = conversationMemory.userTopics
        .slice(0, 3) // Top 3 topics
        .map(topic => `${topic.topic} (${topic.status}, severity: ${topic.severity})`)
        .join(', ');
      contextParts.push(`Ongoing topics: ${topicContext}`);
    }

    // Add session context if available
    if (conversationMemory.sessionContext?.primaryTopic) {
      contextParts.push(`Session focus: ${conversationMemory.sessionContext.primaryTopic}`);
    }

    return contextParts.join('\n\n');
  }

  private calculateRelationshipStrength(aiResponse: any): number {
    // Calculate relationship strength based on response quality and cultural awareness
    const baseStrength = 0.5;
    const qualityBonus = (aiResponse.quality_metrics?.empathy_score || 0) * 0.3;
    const culturalBonus = (aiResponse.quality_metrics?.cultural_awareness_score || 0) * 0.2;

    return Math.min(baseStrength + qualityBonus + culturalBonus, 1.0);
  }

  private extractImprovementAreas(aiResponse: any): string[] {
    const improvements: string[] = [];

    // If mood is positive or neutral, consider suggested resources as improvement areas
    if (aiResponse.metadata?.mood_detected === 'positive' || aiResponse.metadata?.mood_detected === 'neutral') {
      improvements.push(...(aiResponse.suggested_resources || []));
    }

    return improvements;
  }

  private extractConcernAreas(aiResponse: any): string[] {
    const concerns: string[] = [];

    // If mood is negative with high intensity, mark suggested resources as concerns
    if (aiResponse.metadata?.mood_detected === 'negative' &&
        (aiResponse.metadata?.emotional_intensity || 0) >= 7) {
      concerns.push(...(aiResponse.suggested_resources || []));
    }

    return concerns;
  }

  private async updateUserTopics(
    userId: string,
    detectedTopics: string[],
  ) {
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
}
