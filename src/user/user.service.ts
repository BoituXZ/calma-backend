import { Injectable, NotFoundException } from '@nestjs/common';
import { DatabaseService } from '../database/database.service';

export interface UpdateUserDto {
  name?: string;
  email?: string;
}

@Injectable()
export class UserService {
  constructor(private database: DatabaseService) {}

  async getUserProfile(userId: string) {
    const user = await this.database.user.findUnique({
      where: { id: userId },
      select: {
        id: true,
        name: true,
        email: true,
        role: true,
        createdAt: true,
      },
    });

    if (!user) {
      throw new NotFoundException('User not found');
    }

    return user;
  }

  async updateUserProfile(userId: string, updateData: UpdateUserDto) {
    try {
      const user = await this.database.user.update({
        where: { id: userId },
        data: updateData,
        select: {
          id: true,
          name: true,
          email: true,
          role: true,
          createdAt: true,
        },
      });

      return user;
    } catch (error) {
      throw new NotFoundException('User not found');
    }
  }

  async deleteUser(userId: string) {
    try {
      await this.database.user.delete({
        where: { id: userId },
      });

      return { message: 'User deleted successfully' };
    } catch (error) {
      throw new NotFoundException('User not found');
    }
  }

  async getAllTherapists() {
    const therapists = await this.database.user.findMany({
      where: {
        role: 'THERAPIST',
      },
      select: {
        id: true,
        name: true,
        email: true,
        role: true,
        createdAt: true,
      },
      orderBy: {
        name: 'asc',
      },
    });

    return therapists;
  }
}
