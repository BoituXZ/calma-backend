import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { AuthModule } from './auth/auth.module';
import { UserModule } from './user/user.module';
import { MoodModule } from './mood/mood.module';
import { ChatModule } from './chat/chat.module';
import { TherapistChatModule } from './therapist-chat/therapist-chat.module';
import { ResourcesModule } from './resources/resources.module';
import { SavedResourceModule } from './saved-resource/saved-resource.module';
import { DatabaseModule } from './database/database.module';
import { CulturalProfileModule } from './cultural-profile/cultural-profile.module';
import { ResourcesController } from './resources/resources.controller';
import { ResourcesService } from './resources/resources.service';

@Module({
  imports: [
    AuthModule,
    UserModule,
    MoodModule,
    ChatModule,
    TherapistChatModule,
    ResourcesModule,
    SavedResourceModule,
    DatabaseModule,
    CulturalProfileModule,
  ],
  controllers: [AppController, ResourcesController],
  providers: [AppService, ResourcesService],
})
export class AppModule {}
