import { Body, Controller, Post } from '@nestjs/common';
import { ChatService } from './chat.service';
import { ChatRequestDto, ChatResponseDto } from '../common/dto/chat.dto';

@Controller('chat')
export class ChatController {
  constructor(private readonly chatService: ChatService) {}

  @Post('message')
  async sendMessage(
    @Body() chatRequest: ChatRequestDto,
  ): Promise<ChatResponseDto> {
    return this.chatService.handleMessage(
      chatRequest.userId,
      chatRequest.message,
      chatRequest.sessionId,
    );
  }
}
