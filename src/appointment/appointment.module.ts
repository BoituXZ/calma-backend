import { Module } from '@nestjs/common';
import { AppointmentService } from './appointment.service';
import { AppointmentController } from './appointment.controller';
import { DatabaseModule } from '../database/database.module';
import { AuthModule } from '../auth/auth.module';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';
import { RolesGuard } from '../common/guards/roles.guard';

@Module({
  imports: [DatabaseModule, AuthModule],
  providers: [AppointmentService, JwtAuthGuard, RolesGuard],
  controllers: [AppointmentController],
  exports: [AppointmentService],
})
export class AppointmentModule {}
