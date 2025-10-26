import {
  Injectable,
  CanActivate,
  ExecutionContext,
  UnauthorizedException,
} from '@nestjs/common';
import { Request } from 'express';
import { AuthService } from '../../auth/auth.service';

@Injectable()
export class JwtAuthGuard implements CanActivate {
  constructor(private authService: AuthService) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const request = context.switchToHttp().getRequest<Request>();
    const token = request.cookies?.jwt;

    if (!token) {
      throw new UnauthorizedException('No authentication token found');
    }

    try {
      const user = await this.authService.verifyToken(token);

      if (!user) {
        throw new UnauthorizedException('Invalid or expired token');
      }

      // Attach user to request for use in controllers
      request['user'] = user;
      return true;
    } catch (error) {
      throw new UnauthorizedException('Authentication failed');
    }
  }
}
