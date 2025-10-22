import { Injectable, NotFoundException } from '@nestjs/common';
import { ResourceType } from '@prisma/client';
import { DatabaseService } from 'src/database/database.service';

export interface ResourceFilters {
  type?: ResourceType;
  tags?: string[];
  culturalTags?: string[];
}

export interface CreateResourceDto {
  title: string;
  description?: string;
  type: ResourceType;
  link: string;
  tags?: string[];
  culturalTags?: string[];
  targetAudience?: string[];
}

@Injectable()
export class ResourcesService {
  constructor(private database: DatabaseService) {}

  async getAllResources(filters?: ResourceFilters) {
    const where: any = {};

    if (filters?.type) {
      where.type = filters.type;
    }

    if (filters?.tags && filters.tags.length > 0) {
      where.tags = {
        hasSome: filters.tags,
      };
    }

    if (filters?.culturalTags && filters.culturalTags.length > 0) {
      where.culturalTags = {
        hasSome: filters.culturalTags,
      };
    }

    const resources = await this.database.resource.findMany({
      where,
      orderBy: { createdAt: 'desc' },
    });

    return resources;
  }

  async getResourceById(id: string) {
    const resource = await this.database.resource.findUnique({
      where: { id },
    });

    if (!resource) {
      throw new NotFoundException('Resource not found');
    }

    return resource;
  }

  async createResource(data: CreateResourceDto) {
    const resource = await this.database.resource.create({
      data: {
        title: data.title,
        description: data.description,
        type: data.type,
        link: data.link,
        tags: data.tags || [],
        culturalTags: data.culturalTags || [],
        targetAudience: data.targetAudience || [],
      },
    });

    return resource;
  }
}
