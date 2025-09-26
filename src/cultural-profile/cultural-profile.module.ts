import { Module } from '@nestjs/common';
import { CulturalProfileController } from './cultural-profile.controller';
import { CulturalProfileService } from './cultural-profile.service';
import { DatabaseService } from '../database/database.service';

@Module({
  controllers: [CulturalProfileController],
  providers: [CulturalProfileService, DatabaseService],
  exports: [CulturalProfileService],
})
export class CulturalProfileModule {}
