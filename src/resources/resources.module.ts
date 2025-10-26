import { Module } from '@nestjs/common';
import { ResourcesController } from './resources.controller';
import { ResourcesService } from './resources.service';
import { DatabaseModule } from '../database/database.module';
import { AuthModule } from '../auth/auth.module';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';
import { RolesGuard } from '../common/guards/roles.guard';

@Module({
  imports: [DatabaseModule, AuthModule],
  controllers: [ResourcesController],
  providers: [ResourcesService, JwtAuthGuard, RolesGuard],
  exports: [ResourcesService],
})
export class ResourcesModule {}
