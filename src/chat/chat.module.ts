import { Module } from '@nestjs/common';
import { ChatController } from './chat.controller';
import { ChatService } from './chat.service';
import { DatabaseService } from 'src/database/database.service';


@Module({
  controllers: [ChatController],
  providers: [ChatService, DatabaseService],
})
export class ChatModule {}
