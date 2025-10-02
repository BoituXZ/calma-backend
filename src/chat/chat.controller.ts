import {
  Body,
  Controller,
  Post,
  Get,
  Req,
  HttpException,
  HttpStatus,
  Logger
} from '@nestjs/common';
import { Request } from 'express';
import { ChatService } from './chat.service';
import { AuthService } from '../auth/auth.service';
import { ChatRequestDto, ChatResponseDto } from '../common/dto/chat.dto';

@Controller('chat')
export class ChatController {
  private readonly logger = new Logger(ChatController.name);

  constructor(
    private readonly chatService: ChatService,
    private readonly authService: AuthService,
  ) {}

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
    @Req() req: Request,
    @Body() chatRequest: ChatRequestDto,
  ): Promise<ChatResponseDto> {
    try {
      // Extract JWT from cookie
      const token = req.cookies?.jwt;

      if (!token) {
        throw new HttpException(
          'Authentication required',
          HttpStatus.UNAUTHORIZED,
        );
      }

      // Verify token and get user
      const user = await this.authService.verifyToken(token);

      if (!user) {
        throw new HttpException(
          'Invalid or expired token',
          HttpStatus.UNAUTHORIZED,
        );
      }

      this.logger.log(`Processing message for user: ${user.id}`);

      const result = await this.chatService.handleMessage(
        user.id,
        chatRequest.message,
        chatRequest.sessionId,
      );

      this.logger.log(
        `Message processed successfully for user: ${user.id}, session: ${result.session.id}`,
      );

      return result;
    } catch (error) {
      this.logger.error(
        `Failed to process message`,
        error.stack,
      );

      // Re-throw the error to let NestJS handle the HTTP response
      throw error;
    }
  }
}
