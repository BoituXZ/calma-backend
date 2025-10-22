import {
  Body,
  Controller,
  Post,
  Get,
  Req,
  Query,
  HttpException,
  HttpStatus,
  Logger,
  UseGuards,
} from '@nestjs/common';
import { Request } from 'express';
import { ChatService } from './chat.service';
import { AuthService } from '../auth/auth.service';
import { ChatRequestDto, ChatResponseDto } from '../common/dto/chat.dto';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';
import { CurrentUser } from '../common/decorators/user.decorator';
import { ConversationSessionService } from '../conversation-session/conversation-session.service';

@Controller('chat')
export class ChatController {
  private readonly logger = new Logger(ChatController.name);

  constructor(
    private readonly chatService: ChatService,
    private readonly authService: AuthService,
    private readonly sessionService: ConversationSessionService,
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
      this.logger.error(`Failed to process message`, error.stack);

      // Re-throw the error to let NestJS handle the HTTP response
      throw error;
    }
  }

  @Get('history')
  @UseGuards(JwtAuthGuard)
  async getChatHistory(
    @CurrentUser() user: any,
    @Query('limit') limit?: string,
    @Query('sessionId') sessionId?: string,
  ) {
    try {
      const limitNum = limit ? parseInt(limit, 10) : 50;

      if (sessionId) {
        // Get chats for specific session
        const session =
          await this.sessionService.getSessionWithChats(sessionId);

        if (!session || session.userId !== user.id) {
          throw new HttpException(
            'Session not found or access denied',
            HttpStatus.FORBIDDEN,
          );
        }

        return {
          session: {
            id: session.id,
            title: session.title,
            startTime: session.startTime,
            endTime: session.endTime,
          },
          chats: session.chats,
        };
      } else {
        // Get recent chats across all sessions
        const chats = await this.sessionService.getRecentChats(
          user.id,
          limitNum,
        );
        return { chats };
      }
    } catch (error) {
      this.logger.error('Failed to get chat history', error.stack);
      throw error;
    }
  }

  @Get('sessions')
  @UseGuards(JwtAuthGuard)
  async getSessions(@CurrentUser() user: any) {
    try {
      const sessions = await this.chatService.getUserSessions(user.id);
      return { sessions };
    } catch (error) {
      this.logger.error('Failed to get sessions', error.stack);
      throw new HttpException(
        'Failed to retrieve sessions',
        HttpStatus.INTERNAL_SERVER_ERROR,
      );
    }
  }
}
