import { Injectable, NotFoundException } from '@nestjs/common';
import { DatabaseService } from '../database/database.service';
import {
  CreateCulturalProfileDto,
  UpdateCulturalProfileDto,
} from '../common/dto/cultural-profile.dto';

@Injectable()
export class CulturalProfileService {
  constructor(private prisma: DatabaseService) {}

  async create(userId: string, createDto: CreateCulturalProfileDto) {
    return this.prisma.culturalProfile.create({
      data: {
        userId,
        ...createDto,
      },
    });
  }

  async findByUserId(userId: string) {
    const profile = await this.prisma.culturalProfile.findUnique({
      where: { userId },
    });

    if (!profile) {
      throw new NotFoundException(
        `Cultural profile not found for user ${userId}`,
      );
    }

    return profile;
  }

  async update(userId: string, updateDto: UpdateCulturalProfileDto) {
    const existingProfile = await this.prisma.culturalProfile.findUnique({
      where: { userId },
    });

    if (!existingProfile) {
      throw new NotFoundException(
        `Cultural profile not found for user ${userId}`,
      );
    }

    return this.prisma.culturalProfile.update({
      where: { userId },
      data: updateDto,
    });
  }

  async createDefaultProfile(userId: string) {
    return this.prisma.culturalProfile.create({
      data: {
        userId,
        ageGroup: 'ADULT',
        location: 'URBAN',
        educationLevel: 'SECONDARY',
        familyStructure: 'NUCLEAR',
        hasElders: false,
        respectLevel: 'MODERATE',
        economicStatus: 'MIDDLE',
      },
    });
  }

  async getOrCreateProfile(userId: string) {
    try {
      return await this.findByUserId(userId);
    } catch (error) {
      if (error instanceof NotFoundException) {
        return await this.createDefaultProfile(userId);
      }
      throw error;
    }
  }
}
