import {
  Controller,
  Get,
  Put,
  Delete,
  Param,
  Body,
  UseGuards,
  HttpStatus,
  HttpCode,
} from '@nestjs/common';
import { UserService, UpdateUserDto } from './user.service';
import { JwtAuthGuard } from '../common/guards/jwt-auth.guard';
import { RolesGuard } from '../common/guards/roles.guard';
import { Roles } from '../common/decorators/roles.decorator';
import { CurrentUser } from '../common/decorators/user.decorator';
import { Role } from '@prisma/client';

@Controller('user')
@UseGuards(JwtAuthGuard)
export class UserController {
  constructor(private userService: UserService) {}

  @Get('profile')
  async getOwnProfile(@CurrentUser() user: any) {
    return this.userService.getUserProfile(user.id);
  }

  @Get('therapists')
  async getAllTherapists() {
    const therapists = await this.userService.getAllTherapists();
    return { therapists };
  }

  @Get(':id')
  @UseGuards(RolesGuard)
  @Roles(Role.ADMIN)
  async getUserById(@Param('id') id: string) {
    return this.userService.getUserProfile(id);
  }

  @Put('profile')
  async updateOwnProfile(
    @CurrentUser() user: any,
    @Body() updateData: UpdateUserDto,
  ) {
    return this.userService.updateUserProfile(user.id, updateData);
  }

  @Put(':id')
  @UseGuards(RolesGuard)
  @Roles(Role.ADMIN)
  async updateUserById(
    @Param('id') id: string,
    @Body() updateData: UpdateUserDto,
  ) {
    return this.userService.updateUserProfile(id, updateData);
  }

  @Delete('profile')
  @HttpCode(HttpStatus.OK)
  async deleteOwnProfile(@CurrentUser() user: any) {
    return this.userService.deleteUser(user.id);
  }

  @Delete(':id')
  @UseGuards(RolesGuard)
  @Roles(Role.ADMIN)
  @HttpCode(HttpStatus.OK)
  async deleteUserById(@Param('id') id: string) {
    return this.userService.deleteUser(id);
  }
}
