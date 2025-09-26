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

  async handleMessage(userId: string, message: string, sessionId?: string) {
    try {
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

      // Call FastAPI AI service
      const response = await firstValueFrom(
        this.httpService.post<AIServiceResponse>(
          'http://localhost:8000/chat',
          aiRequest,
        ),
      );

      const aiResponse = response.data;

      // Save bot message with AI metadata
      const botMessage = await this.prisma.chat.create({
        data: {
          userId,
          sender: 'BOT',
          message: aiResponse.response,
          sessionId: session.id,
          emotionalTone: aiResponse.emotionalTone,
          detectedTopics: aiResponse.detectedTopics || [],
          culturalContext: aiResponse.culturalAdaptations,
          memoryReferences: [], // Could be populated based on AI response
        },
      });

      // Update session with AI insights
      if (aiResponse.memoryUpdates) {
        await this.sessionService.updateSession(session.id, {
          keyInsights: aiResponse.memoryUpdates.keyInsights,
          actionItems: aiResponse.memoryUpdates.actionItems || [],
        });
      }

      // Update user topics if detected
      if (aiResponse.detectedTopics && aiResponse.detectedTopics.length > 0) {
        await this.updateUserTopics(
          userId,
          aiResponse.detectedTopics,
          aiResponse.memoryUpdates?.topicUpdates,
        );
      }

      // Update progress tracking if provided
      if (aiResponse.progressUpdates) {
        await this.updateProgressTracking(userId, aiResponse.progressUpdates);
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
        },
        session: {
          id: session.id,
          title: session.title || undefined,
        },
      };
    } catch (err) {
      console.error('ChatService error:', err);
      throw new InternalServerErrorException('Failed to process message');
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

  private async updateUserTopics(
    userId: string,
    detectedTopics: string[],
    topicUpdates?: { topic: string; severity?: string; status?: string }[],
  ) {
    for (const topic of detectedTopics) {
      const updateData = topicUpdates?.find((update) => update.topic === topic);

      await this.prisma.userTopic.upsert({
        where: {
          userId_topic: { userId, topic },
        },
        update: {
          frequency: { increment: 1 },
          lastDiscussed: new Date(),
          ...(updateData?.severity && { severity: updateData.severity as any }),
          ...(updateData?.status && { status: updateData.status as any }),
        },
        create: {
          userId,
          topic,
          frequency: 1,
          severity: (updateData?.severity as any) || 'MODERATE',
          status: (updateData?.status as any) || 'ONGOING',
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
