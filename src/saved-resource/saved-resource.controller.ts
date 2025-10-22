import {
  Controller,
  Get,
  Post,
  Delete,
  Param,
  Body,
  UseGuards,
  HttpCode,
  HttpStatus,
} from '@nestjs/common';
import { SavedResourceService } from './saved-resource.service';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';
import { CurrentUser } from '../common/decorators/user.decorator';

export interface SaveResourceDto {
  resourceId: string;
  recommendationReason?: string;
  culturalRelevance?: string;
}

@Controller('saved-resource')
@UseGuards(JwtAuthGuard)
export class SavedResourceController {
  constructor(private savedResourceService: SavedResourceService) {}

  @Post()
  async saveResource(
    @CurrentUser() user: any,
    @Body() saveDto: SaveResourceDto,
  ) {
    const savedResource = await this.savedResourceService.saveResource(
      user.id,
      saveDto,
    );
    return {
      savedResource,
      message: 'Resource saved successfully',
    };
  }

  @Get()
  async getUserSavedResources(@CurrentUser() user: any) {
    const savedResources =
      await this.savedResourceService.getUserSavedResources(user.id);
    return { savedResources };
  }

  @Delete(':id')
  @HttpCode(HttpStatus.OK)
  async unsaveResource(@CurrentUser() user: any, @Param('id') id: string) {
    await this.savedResourceService.unsaveResource(user.id, id);
    return { message: 'Resource removed from saved' };
  }
}
