import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { AuthModule } from './auth/auth.module';
import { UserModule } from './user/user.module';
import { MoodModule } from './mood/mood.module';
import { ChatModule } from './chat/chat.module';
import { TherapistChatModule } from './therapist-chat/therapist-chat.module';
import { ResourceModule } from './resource/resource.module';
import { SavedResourceModule } from './saved-resource/saved-resource.module';

@Module({
  imports: [AuthModule, UserModule, MoodModule, ChatModule, TherapistChatModule, ResourceModule, SavedResourceModule],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
