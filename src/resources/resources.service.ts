import { Injectable } from '@nestjs/common';
import { Prisma } from '@prisma/client';
import { UserDto } from 'src/common/dto/user.dto';
import { DatabaseService } from 'src/database/database.service';

@Injectable()
export class ResourcesService {
  constructor(private database: DatabaseService) {}

  async getAllResources() {
    const resources = this.database.resource.findMany();

    return resources;
  }
}
