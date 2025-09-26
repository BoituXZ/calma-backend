import { Module } from '@nestjs/common';
import { SavedResourceService } from './saved-resource.service';
import { SavedResourceController } from './saved-resource.controller';

@Module({
  providers: [SavedResourceService],
  controllers: [SavedResourceController],
})
export class SavedResourceModule {}
