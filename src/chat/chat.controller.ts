import {
  Body,
  Controller,
  Post,
  Get,
  HttpException,
  HttpStatus,
  Logger
} from '@nestjs/common';
import { ChatService } from './chat.service';
import { ChatRequestDto, ChatResponseDto } from '../common/dto/chat.dto';

@Controller('chat')
export class ChatController {
  private readonly logger = new Logger(ChatController.name);

  constructor(private readonly chatService: ChatService) {}

  @Get('health')
  async checkHealth() {
    try {
      const isHealthy = await this.chatService.checkAIServiceHealth();
      return {
        status: isHealthy ? 'healthy' : 'unhealthy',
        ai_service: isHealthy ? 'available' : 'unavailable',
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      this.logger.error('Health check failed:', error);
      throw new HttpException(
        {
          status: 'error',
          message: 'Health check failed',
          timestamp: new Date().toISOString(),
        },
        HttpStatus.SERVICE_UNAVAILABLE,
      );
    }
  }

  @Post('message')
  async sendMessage(
    @Body() chatRequest: ChatRequestDto,
  ): Promise<ChatResponseDto> {
    try {
      this.logger.log(`Processing message for user: ${chatRequest.userId}`);

      const result = await this.chatService.handleMessage(
        chatRequest.userId,
        chatRequest.message,
        chatRequest.sessionId,
      );

      this.logger.log(
        `Message processed successfully for user: ${chatRequest.userId}, session: ${result.session.id}`,
      );

      return result;
    } catch (error) {
      this.logger.error(
        `Failed to process message for user: ${chatRequest.userId}`,
        error.stack,
      );

      // Re-throw the error to let NestJS handle the HTTP response
      throw error;
    }
  }
}
