import { Module } from '@nestjs/common';
import { HttpModule } from '@nestjs/axios';
import { ChatController } from './chat.controller';
import { ChatService } from './chat.service';
import { DatabaseService } from '../database/database.service';
import { CulturalProfileModule } from '../cultural-profile/cultural-profile.module';
import { ConversationSessionService } from '../conversation-session/conversation-session.service';
import { AuthModule } from '../auth/auth.module';

@Module({
  imports: [HttpModule, CulturalProfileModule, AuthModule],
  controllers: [ChatController],
  providers: [ChatService, DatabaseService, ConversationSessionService],
  exports: [ChatService],
})
export class ChatModule {}
