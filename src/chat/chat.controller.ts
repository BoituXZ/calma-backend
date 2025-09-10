import { Body, Controller, Post } from '@nestjs/common';
import { ChatService } from './chat.service';

@Controller('chat')
export class ChatController {
  constructor(private readonly chatService: ChatService) {}

  @Post('message')
  async sendMessage(
    @Body('userId') userId: string,
    @Body('message') message: string,
  ) {
    return this.chatService.handleMessage(userId, message);
  }
}
