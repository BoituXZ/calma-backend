import {
  Controller,
  Get,
  Post,
  Put,
  Delete,
  Body,
  Param,
  UseGuards,
  HttpCode,
  HttpStatus,
} from '@nestjs/common';
import {
  AppointmentService,
  CreateAppointmentDto,
  UpdateAppointmentDto,
} from './appointment.service';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';
import { RolesGuard } from '../common/guards/roles.guard';
import { Roles } from '../common/decorators/roles.decorator';
import { CurrentUser } from '../common/decorators/user.decorator';
import { Role } from '@prisma/client';

@Controller('appointments')
@UseGuards(JwtAuthGuard)
export class AppointmentController {
  constructor(private appointmentService: AppointmentService) {}

  @Post()
  async createAppointment(
    @CurrentUser() user: any,
    @Body() createDto: CreateAppointmentDto,
  ) {
    const appointment = await this.appointmentService.createAppointment(
      user.id,
      createDto,
    );
    return {
      appointment,
      message: 'Appointment created successfully',
    };
  }

  @Get('user')
  async getUserAppointments(@CurrentUser() user: any) {
    const appointments = await this.appointmentService.getUserAppointments(
      user.id,
    );
    return { appointments };
  }

  @Get('therapist')
  @UseGuards(RolesGuard)
  @Roles(Role.THERAPIST)
  async getTherapistAppointments(@CurrentUser() user: any) {
    const appointments = await this.appointmentService.getTherapistAppointments(
      user.id,
    );
    return { appointments };
  }

  @Get(':id')
  async getAppointmentById(
    @CurrentUser() user: any,
    @Param('id') appointmentId: string,
  ) {
    const appointment = await this.appointmentService.getAppointmentById(
      appointmentId,
      user.id,
    );
    return { appointment };
  }

  @Put(':id')
  async updateAppointment(
    @CurrentUser() user: any,
    @Param('id') appointmentId: string,
    @Body() updateDto: UpdateAppointmentDto,
  ) {
    const appointment = await this.appointmentService.updateAppointment(
      appointmentId,
      user.id,
      updateDto,
    );
    return {
      appointment,
      message: 'Appointment updated successfully',
    };
  }

  @Delete(':id')
  @HttpCode(HttpStatus.OK)
  async cancelAppointment(
    @CurrentUser() user: any,
    @Param('id') appointmentId: string,
  ) {
    const appointment = await this.appointmentService.cancelAppointment(
      appointmentId,
      user.id,
    );
    return {
      appointment,
      message: 'Appointment cancelled successfully',
    };
  }
}
