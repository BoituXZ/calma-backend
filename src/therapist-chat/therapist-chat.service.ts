import { Injectable, NotFoundException } from '@nestjs/common';
import { DatabaseService } from '../database/database.service';
import { Sender } from '@prisma/client';

export interface SendMessageDto {
  therapistId: string;
  message: string;
}

@Injectable()
export class TherapistChatService {
  constructor(private database: DatabaseService) {}

  async sendMessage(userId: string, sendMessageDto: SendMessageDto) {
    const { therapistId, message } = sendMessageDto;

    // Verify therapist exists and has THERAPIST role
    const therapist = await this.database.user.findUnique({
      where: { id: therapistId },
    });

    if (!therapist || therapist.role !== 'THERAPIST') {
      throw new NotFoundException('Therapist not found');
    }

    // Create the message
    const chatMessage = await this.database.therapistChat.create({
      data: {
        userId,
        therapistId,
        message,
        sender: Sender.USER,
      },
    });

    return chatMessage;
  }

  async sendTherapistMessage(
    therapistId: string,
    userId: string,
    message: string,
  ) {
    // Verify user exists
    const user = await this.database.user.findUnique({
      where: { id: userId },
    });

    if (!user) {
      throw new NotFoundException('User not found');
    }

    const chatMessage = await this.database.therapistChat.create({
      data: {
        userId,
        therapistId,
        message,
        sender: Sender.THERAPIST,
      },
    });

    return chatMessage;
  }

  async getConversation(userId: string, therapistId: string) {
    const messages = await this.database.therapistChat.findMany({
      where: {
        userId,
        therapistId,
      },
      orderBy: { timestamp: 'asc' },
      include: {
        user: {
          select: {
            id: true,
            name: true,
            email: true,
          },
        },
      },
    });

    return messages;
  }

  async getUserConversations(userId: string) {
    // Get all unique therapists the user has chatted with
    const conversations = await this.database.therapistChat.findMany({
      where: { userId },
      select: {
        therapistId: true,
        timestamp: true,
      },
      orderBy: { timestamp: 'desc' },
    });

    // Get unique therapist IDs
    const therapistIds = [...new Set(conversations.map((c) => c.therapistId))];

    // Get therapist details and last message for each
    const conversationDetails = await Promise.all(
      therapistIds.map(async (therapistId) => {
        const therapist = await this.database.user.findUnique({
          where: { id: therapistId },
          select: { id: true, name: true, email: true },
        });

        const lastMessage = await this.database.therapistChat.findFirst({
          where: { userId, therapistId },
          orderBy: { timestamp: 'desc' },
        });

        const messageCount = await this.database.therapistChat.count({
          where: { userId, therapistId },
        });

        return {
          therapist,
          lastMessage,
          messageCount,
        };
      }),
    );

    return conversationDetails;
  }

  async getTherapistConversations(therapistId: string) {
    // Get all unique users the therapist has chatted with
    const conversations = await this.database.therapistChat.findMany({
      where: { therapistId },
      select: {
        userId: true,
        timestamp: true,
      },
      orderBy: { timestamp: 'desc' },
    });

    // Get unique user IDs
    const userIds = [...new Set(conversations.map((c) => c.userId))];

    // Get user details and last message for each
    const conversationDetails = await Promise.all(
      userIds.map(async (userId) => {
        const user = await this.database.user.findUnique({
          where: { id: userId },
          select: { id: true, name: true, email: true },
        });

        const lastMessage = await this.database.therapistChat.findFirst({
          where: { userId, therapistId },
          orderBy: { timestamp: 'desc' },
        });

        const messageCount = await this.database.therapistChat.count({
          where: { userId, therapistId },
        });

        return {
          user,
          lastMessage,
          messageCount,
        };
      }),
    );

    return conversationDetails;
  }
}
