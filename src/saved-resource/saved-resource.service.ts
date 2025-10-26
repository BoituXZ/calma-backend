import {
  Injectable,
  NotFoundException,
  ConflictException,
} from '@nestjs/common';
import { DatabaseService } from '../database/database.service';

export interface SaveResourceDto {
  resourceId: string;
  recommendationReason?: string;
  culturalRelevance?: string;
}

@Injectable()
export class SavedResourceService {
  constructor(private database: DatabaseService) {}

  async saveResource(userId: string, saveDto: SaveResourceDto) {
    // Check if resource exists
    const resource = await this.database.resource.findUnique({
      where: { id: saveDto.resourceId },
    });

    if (!resource) {
      throw new NotFoundException('Resource not found');
    }

    // Check if already saved
    const existing = await this.database.savedResource.findFirst({
      where: {
        userId,
        resourceId: saveDto.resourceId,
      },
    });

    if (existing) {
      throw new ConflictException('Resource already saved');
    }

    const savedResource = await this.database.savedResource.create({
      data: {
        userId,
        resourceId: saveDto.resourceId,
        recommendationReason: saveDto.recommendationReason,
        culturalRelevance: saveDto.culturalRelevance,
      },
      include: {
        resource: true,
      },
    });

    return savedResource;
  }

  async getUserSavedResources(userId: string) {
    const savedResources = await this.database.savedResource.findMany({
      where: { userId },
      include: {
        resource: true,
      },
      orderBy: { savedAt: 'desc' },
    });

    return savedResources;
  }

  async unsaveResource(userId: string, savedResourceId: string) {
    const savedResource = await this.database.savedResource.findFirst({
      where: {
        id: savedResourceId,
        userId,
      },
    });

    if (!savedResource) {
      throw new NotFoundException('Saved resource not found');
    }

    await this.database.savedResource.delete({
      where: { id: savedResourceId },
    });

    return { message: 'Resource removed successfully' };
  }
}
