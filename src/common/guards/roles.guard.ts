import {
  Injectable,
  CanActivate,
  ExecutionContext,
  ForbiddenException,
} from '@nestjs/common';
import { Reflector } from '@nestjs/core';
import { Role } from '@prisma/client';
import { Request } from 'express';
import { DatabaseService } from '../../database/database.service';

@Injectable()
export class RolesGuard implements CanActivate {
  constructor(
    private reflector: Reflector,
    private database: DatabaseService,
  ) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const requiredRoles = this.reflector.getAllAndOverride<Role[]>('roles', [
      context.getHandler(),
      context.getClass(),
    ]);

    if (!requiredRoles || requiredRoles.length === 0) {
      return true; // No roles required, allow access
    }

    const request = context.switchToHttp().getRequest<Request>();
    const user = request['user'];

    if (!user) {
      throw new ForbiddenException('User not authenticated');
    }

    // Fetch full user details including role
    const fullUser = await this.database.user.findUnique({
      where: { id: user.id },
      select: { id: true, role: true },
    });

    if (!fullUser) {
      throw new ForbiddenException('User not found');
    }

    const hasRole = requiredRoles.includes(fullUser.role);

    if (!hasRole) {
      throw new ForbiddenException(
        `Access denied. Required role: ${requiredRoles.join(' or ')}`,
      );
    }

    // Attach role to user object for convenience
    request['user'].role = fullUser.role;

    return true;
  }
}
