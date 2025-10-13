import { Module } from '@nestjs/common';
import { TherapistChatService } from './therapist-chat.service';
import { TherapistChatController } from './therapist-chat.controller';

@Module({
  providers: [TherapistChatService],
  controllers: [TherapistChatController],
})
export class TherapistChatModule {}
