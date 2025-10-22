import {
  Controller,
  Get,
  Post,
  Body,
  Param,
  Query,
  UseGuards,
} from '@nestjs/common';
import { TherapistChatService, SendMessageDto } from './therapist-chat.service';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';
import { RolesGuard } from '../common/guards/roles.guard';
import { Roles } from '../common/decorators/roles.decorator';
import { CurrentUser } from '../common/decorators/user.decorator';
import { Role } from '@prisma/client';

@Controller('therapist-chat')
@UseGuards(JwtAuthGuard)
export class TherapistChatController {
  constructor(private therapistChatService: TherapistChatService) {}

  // User sends message to therapist
  @Post('message')
  async sendMessage(
    @CurrentUser() user: any,
    @Body() sendMessageDto: SendMessageDto,
  ) {
    const message = await this.therapistChatService.sendMessage(
      user.id,
      sendMessageDto,
    );
    return {
      message,
      success: true,
    };
  }

  // Therapist sends message to user
  @Post('therapist/message')
  @UseGuards(RolesGuard)
  @Roles(Role.THERAPIST)
  async sendTherapistMessage(
    @CurrentUser() user: any,
    @Body() body: { userId: string; message: string },
  ) {
    const message = await this.therapistChatService.sendTherapistMessage(
      user.id,
      body.userId,
      body.message,
    );
    return {
      message,
      success: true,
    };
  }

  // Get conversation between user and specific therapist
  @Get('conversation/:therapistId')
  async getConversation(
    @CurrentUser() user: any,
    @Param('therapistId') therapistId: string,
  ) {
    const messages = await this.therapistChatService.getConversation(
      user.id,
      therapistId,
    );
    return { messages };
  }

  // User gets all their conversations
  @Get('conversations')
  async getUserConversations(@CurrentUser() user: any) {
    const conversations = await this.therapistChatService.getUserConversations(
      user.id,
    );
    return { conversations };
  }

  // Therapist gets all their conversations
  @Get('therapist/conversations')
  @UseGuards(RolesGuard)
  @Roles(Role.THERAPIST)
  async getTherapistConversations(@CurrentUser() user: any) {
    const conversations =
      await this.therapistChatService.getTherapistConversations(user.id);
    return { conversations };
  }

  // Admin endpoint to get specific conversation
  @Get('admin/conversation')
  @UseGuards(RolesGuard)
  @Roles(Role.ADMIN)
  async getSpecificConversation(
    @Query('userId') userId: string,
    @Query('therapistId') therapistId: string,
  ) {
    const messages = await this.therapistChatService.getConversation(
      userId,
      therapistId,
    );
    return { messages };
  }
}
