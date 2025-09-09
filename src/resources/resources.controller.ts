import { Controller, Get, Query } from '@nestjs/common';
import { Prisma } from '@prisma/client';
import { UserDto } from 'src/common/dto/user.dto';
import { ResourcesService } from './resources.service';

@Controller('resources')
export class ResourcesController {

constructor(
    private resourcesService: ResourcesService
){}

@Get('/:userid')
async getResources(@Query() userId:UserDto){
    const response = await this.resourcesService.getAllResources()
}
}
