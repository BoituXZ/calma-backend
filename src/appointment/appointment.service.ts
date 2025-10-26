import {
  Injectable,
  NotFoundException,
  ConflictException,
  BadRequestException,
} from '@nestjs/common';
import { DatabaseService } from '../database/database.service';
import { AppointmentStatus } from '@prisma/client';

export interface CreateAppointmentDto {
  therapistId: string;
  scheduledAt: Date | string;
  duration?: number;
  reason?: string;
  notes?: string;
  meetingLink?: string;
  location?: string;
}

export interface UpdateAppointmentDto {
  scheduledAt?: Date | string;
  duration?: number;
  status?: AppointmentStatus;
  reason?: string;
  notes?: string;
  meetingLink?: string;
  location?: string;
}

@Injectable()
export class AppointmentService {
  constructor(private database: DatabaseService) {}

  async createAppointment(userId: string, createDto: CreateAppointmentDto) {
    // Verify therapist exists and has THERAPIST role
    const therapist = await this.database.user.findUnique({
      where: { id: createDto.therapistId },
    });

    if (!therapist || therapist.role !== 'THERAPIST') {
      throw new NotFoundException('Therapist not found');
    }

    // Check if therapist is available at the requested time
    const conflictingAppointment = await this.checkTherapistAvailability(
      createDto.therapistId,
      new Date(createDto.scheduledAt),
      createDto.duration || 60,
    );

    if (conflictingAppointment) {
      throw new ConflictException(
        'Therapist is not available at the requested time',
      );
    }

    const appointment = await this.database.appointment.create({
      data: {
        userId,
        therapistId: createDto.therapistId,
        scheduledAt: new Date(createDto.scheduledAt),
        duration: createDto.duration || 60,
        reason: createDto.reason,
        notes: createDto.notes,
        meetingLink: createDto.meetingLink,
        location: createDto.location,
      },
      include: {
        user: {
          select: {
            id: true,
            name: true,
            email: true,
          },
        },
      },
    });

    return appointment;
  }

  async getUserAppointments(userId: string) {
    const appointments = await this.database.appointment.findMany({
      where: { userId },
      orderBy: { scheduledAt: 'desc' },
    });

    return appointments;
  }

  async getTherapistAppointments(therapistId: string) {
    const appointments = await this.database.appointment.findMany({
      where: { therapistId },
      orderBy: { scheduledAt: 'asc' },
      include: {
        user: {
          select: {
            id: true,
            name: true,
            email: true,
          },
        },
      },
    });

    return appointments;
  }

  async getAppointmentById(appointmentId: string, userId: string) {
    const appointment = await this.database.appointment.findUnique({
      where: { id: appointmentId },
      include: {
        user: {
          select: {
            id: true,
            name: true,
            email: true,
          },
        },
      },
    });

    if (!appointment) {
      throw new NotFoundException('Appointment not found');
    }

    // Check if user is authorized to view this appointment
    if (appointment.userId !== userId && appointment.therapistId !== userId) {
      throw new BadRequestException('Unauthorized to view this appointment');
    }

    return appointment;
  }

  async updateAppointment(
    appointmentId: string,
    userId: string,
    updateDto: UpdateAppointmentDto,
  ) {
    const appointment = await this.database.appointment.findUnique({
      where: { id: appointmentId },
    });

    if (!appointment) {
      throw new NotFoundException('Appointment not found');
    }

    // Check authorization
    if (appointment.userId !== userId && appointment.therapistId !== userId) {
      throw new BadRequestException('Unauthorized to update this appointment');
    }

    // If rescheduling, check availability
    if (updateDto.scheduledAt) {
      const conflictingAppointment = await this.checkTherapistAvailability(
        appointment.therapistId,
        new Date(updateDto.scheduledAt),
        updateDto.duration || appointment.duration,
        appointmentId,
      );

      if (conflictingAppointment) {
        throw new ConflictException(
          'Therapist is not available at the requested time',
        );
      }
    }

    const updatedAppointment = await this.database.appointment.update({
      where: { id: appointmentId },
      data: {
        ...(updateDto.scheduledAt && {
          scheduledAt: new Date(updateDto.scheduledAt),
        }),
        ...(updateDto.duration && { duration: updateDto.duration }),
        ...(updateDto.status && { status: updateDto.status }),
        ...(updateDto.reason !== undefined && { reason: updateDto.reason }),
        ...(updateDto.notes !== undefined && { notes: updateDto.notes }),
        ...(updateDto.meetingLink !== undefined && {
          meetingLink: updateDto.meetingLink,
        }),
        ...(updateDto.location !== undefined && {
          location: updateDto.location,
        }),
      },
      include: {
        user: {
          select: {
            id: true,
            name: true,
            email: true,
          },
        },
      },
    });

    return updatedAppointment;
  }

  async cancelAppointment(appointmentId: string, userId: string) {
    const appointment = await this.database.appointment.findUnique({
      where: { id: appointmentId },
    });

    if (!appointment) {
      throw new NotFoundException('Appointment not found');
    }

    // Check authorization
    if (appointment.userId !== userId && appointment.therapistId !== userId) {
      throw new BadRequestException('Unauthorized to cancel this appointment');
    }

    const cancelledAppointment = await this.database.appointment.update({
      where: { id: appointmentId },
      data: { status: AppointmentStatus.CANCELLED },
    });

    return cancelledAppointment;
  }

  private async checkTherapistAvailability(
    therapistId: string,
    scheduledAt: Date,
    duration: number,
    excludeAppointmentId?: string,
  ): Promise<boolean> {
    const endTime = new Date(scheduledAt.getTime() + duration * 60000);

    const conflictingAppointments = await this.database.appointment.findMany({
      where: {
        therapistId,
        id: excludeAppointmentId ? { not: excludeAppointmentId } : undefined,
        status: {
          in: [AppointmentStatus.SCHEDULED, AppointmentStatus.CONFIRMED],
        },
        OR: [
          {
            AND: [
              { scheduledAt: { lte: scheduledAt } },
              {
                scheduledAt: {
                  gte: new Date(scheduledAt.getTime() - 60 * 60000),
                },
              },
            ],
          },
          {
            AND: [
              { scheduledAt: { gte: scheduledAt } },
              { scheduledAt: { lt: endTime } },
            ],
          },
        ],
      },
    });

    return conflictingAppointments.length > 0;
  }
}
