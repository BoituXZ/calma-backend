import {
  Controller,
  Get,
  Post,
  Body,
  Param,
  Query,
  UseGuards,
} from '@nestjs/common';
import { ResourcesService } from './resources.service';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';
import { RolesGuard } from '../common/guards/roles.guard';
import { Roles } from '../common/decorators/roles.decorator';
import { Role, ResourceType } from '@prisma/client';

export interface CreateResourceDto {
  title: string;
  description?: string;
  type: ResourceType;
  link: string;
  tags?: string[];
  culturalTags?: string[];
  targetAudience?: string[];
}

@Controller('resources')
export class ResourcesController {
  constructor(private resourcesService: ResourcesService) {}

  @Get()
  async getAllResources(
    @Query('type') type?: ResourceType,
    @Query('tags') tags?: string,
    @Query('culturalTags') culturalTags?: string,
  ) {
    const tagArray = tags ? tags.split(',') : undefined;
    const culturalTagArray = culturalTags ? culturalTags.split(',') : undefined;

    const resources = await this.resourcesService.getAllResources({
      type,
      tags: tagArray,
      culturalTags: culturalTagArray,
    });

    return { resources };
  }

  @Get(':id')
  async getResourceById(@Param('id') id: string) {
    const resource = await this.resourcesService.getResourceById(id);
    return { resource };
  }

  @Post()
  @UseGuards(JwtAuthGuard, RolesGuard)
  @Roles(Role.ADMIN, Role.THERAPIST)
  async createResource(@Body() createDto: CreateResourceDto) {
    const resource = await this.resourcesService.createResource(createDto);
    return { resource, message: 'Resource created successfully' };
  }
}
