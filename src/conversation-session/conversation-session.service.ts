import { Injectable } from '@nestjs/common';
import { DatabaseService } from '../database/database.service';

@Injectable()
export class ConversationSessionService {
  constructor(private prisma: DatabaseService) {}

  async createSession(userId: string) {
    return this.prisma.conversationSession.create({
      data: {
        userId,
        startTime: new Date(),
      },
    });
  }

  async findActiveSession(userId: string) {
    return this.prisma.conversationSession.findFirst({
      where: {
        userId,
        endTime: null,
      },
      orderBy: {
        startTime: 'desc',
      },
    });
  }

  async getOrCreateActiveSession(userId: string) {
    let session = await this.findActiveSession(userId);

    if (!session) {
      session = await this.createSession(userId);
    }

    return session;
  }

  async endSession(
    sessionId: string,
    updates?: {
      title?: string;
      primaryTopic?: string;
      emotionalState?: string;
      keyInsights?: any;
      actionItems?: string[];
    },
  ) {
    return this.prisma.conversationSession.update({
      where: { id: sessionId },
      data: {
        endTime: new Date(),
        ...updates,
      },
    });
  }

  async updateSession(
    sessionId: string,
    updates: {
      title?: string;
      primaryTopic?: string;
      emotionalState?: string;
      culturalFactors?: string[];
      keyInsights?: any;
      actionItems?: string[];
      followUpNeeded?: boolean;
    },
  ) {
    return this.prisma.conversationSession.update({
      where: { id: sessionId },
      data: updates,
    });
  }

  async getSessionWithChats(sessionId: string) {
    return this.prisma.conversationSession.findUnique({
      where: { id: sessionId },
      include: {
        chats: {
          orderBy: { timestamp: 'asc' },
          take: 20,
        },
        topics: {
          include: {
            topic: true,
          },
        },
      },
    });
  }

  async getRecentChats(userId: string, limit: number = 10) {
    return this.prisma.chat.findMany({
      where: { userId },
      orderBy: { timestamp: 'desc' },
      take: limit,
      include: {
        session: true,
      },
    });
  }

  async getUserTopics(userId: string) {
    return this.prisma.userTopic.findMany({
      where: { userId },
      orderBy: { lastDiscussed: 'desc' },
    });
  }

  async getLatestProgressTracking(userId: string) {
    return this.prisma.progressTracking.findFirst({
      where: { userId },
      orderBy: { date: 'desc' },
    });
  }
}
