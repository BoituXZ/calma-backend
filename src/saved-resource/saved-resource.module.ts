import { Module } from '@nestjs/common';
import { SavedResourceService } from './saved-resource.service';
import { SavedResourceController } from './saved-resource.controller';
import { DatabaseModule } from '../database/database.module';
import { AuthModule } from '../auth/auth.module';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';

@Module({
  imports: [DatabaseModule, AuthModule],
  providers: [SavedResourceService, JwtAuthGuard],
  controllers: [SavedResourceController],
  exports: [SavedResourceService],
})
export class SavedResourceModule {}
