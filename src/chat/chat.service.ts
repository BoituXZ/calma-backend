import { Injectable, InternalServerErrorException } from '@nestjs/common';
import { DatabaseService } from 'src/database/database.service';

import axios from 'axios';

@Injectable()
export class ChatService {
  constructor(private prisma: DatabaseService) {}

  async handleMessage(userId: string, message: string) {
    try {
      // Save user message
      const userMessage = await this.prisma.chat.create({
        data: {
          userId,
          sender: 'USER',
          message,
        },
      });

      // Call AI server (dummy for now)
      const aiResponse = await axios.post('http://localhost:5000/ai/respond', {
        message,
        userId,
      });

      // Save bot message
      const botMessage = await this.prisma.chat.create({
        data: {
          userId,
          sender: 'BOT',
          message: aiResponse.data.response,
        },
      });

      return { userMessage, botMessage };
    } catch (err) {
      console.error('ChatService error:', err);
      throw new InternalServerErrorException('Failed to process message');
    }
  }
}
