import { Controller, Get, Post, Put, Body, Param } from '@nestjs/common';
import { CulturalProfileService } from './cultural-profile.service';
import {
  CreateCulturalProfileDto,
  UpdateCulturalProfileDto,
} from '../common/dto/cultural-profile.dto';

@Controller('cultural-profile')
export class CulturalProfileController {
  constructor(
    private readonly culturalProfileService: CulturalProfileService,
  ) {}

  @Post(':userId')
  async create(
    @Param('userId') userId: string,
    @Body() createDto: CreateCulturalProfileDto,
  ) {
    return this.culturalProfileService.create(userId, createDto);
  }

  @Get(':userId')
  async findByUserId(@Param('userId') userId: string) {
    return this.culturalProfileService.findByUserId(userId);
  }

  @Put(':userId')
  async update(
    @Param('userId') userId: string,
    @Body() updateDto: UpdateCulturalProfileDto,
  ) {
    return this.culturalProfileService.update(userId, updateDto);
  }

  @Post(':userId/default')
  async createDefault(@Param('userId') userId: string) {
    return this.culturalProfileService.createDefaultProfile(userId);
  }
}
