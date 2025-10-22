import {
  BadRequestException,
  Injectable,
  InternalServerErrorException,
  UnauthorizedException,
} from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { DatabaseService } from 'src/database/database.service';
import { SignupDto } from 'src/common/dto/signup.dto';
import { LoginDto } from 'src/common/dto/login.dto';
import * as bcrypt from 'bcrypt';

@Injectable()
export class AuthService {
  constructor(
    private database: DatabaseService,
    private jwtService: JwtService,
  ) {}

  async signup(dto: SignupDto) {
    const isExists = await this.database.user.findFirst({
      where: { email: dto.email },
    });

    if (isExists) {
      throw new BadRequestException('Email is already registered');
    }

    // Security check: Prevent ADMIN role creation via signup
    if (dto.role === 'ADMIN') {
      throw new BadRequestException('Admin accounts cannot be created via signup');
    }

    // Default to USER role if not provided
    const userRole = dto.role || 'USER';

    const hash = await bcrypt.hash(dto.password, 10);

    try {
      const user = await this.database.user.create({
        data: {
          name: dto.name,
          email: dto.email,
          password: hash,
          role: userRole,
        },
        select: {
          id: true,
          name: true,
          email: true,
          role: true,
        },
      });

      const token = await this.jwtService.signAsync({
        sub: user.id,
        email: user.email,
      });

      return {
        message: 'User created successfully',
        access_token: token,
        user: {
          id: user.id,
          name: user.name,
          email: user.email,
          role: user.role,
        },
      };
    } catch (error) {
      throw new InternalServerErrorException('Could not create user');
    }
  }

  async login(dto: LoginDto) {
    const { email, password } = dto;

    try {
      const user = await this.database.user.findUnique({
        where: { email },
        select: {
          id: true,
          name: true,
          email: true,
          password: true,
          role: true,
        },
      });

      if (!user) {
        return { message: 'Login failed: wrong email or password' };
      }

      const isPasswordCorrect = await bcrypt.compare(password, user.password);

      if (!isPasswordCorrect) {
        return { message: 'Login failed: wrong email or password' };
      }

      const token = await this.jwtService.signAsync(
        {
          sub: user.id,
          email: user.email,
        },
        { expiresIn: '7d' },
      );

      return {
        message: 'Logged in successfully',
        access_token: token,
        user: {
          id: user.id,
          name: user.name,
          email: user.email,
          role: user.role,
        },
      };
    } catch (error) {
      console.error('Login error:', error);
      throw new InternalServerErrorException('Something went wrong');
    }
  }

  // Add this new method for token verification with debug logging
  async verifyToken(token: string) {
    try {
      console.log('Verifying token...');

      // Verify and decode the JWT token
      const decoded = await this.jwtService.verifyAsync(token);
      console.log('Token decoded successfully:', {
        sub: decoded.sub,
        email: decoded.email,
      });

      // Get user from database using the user ID from token
      const user = await this.database.user.findUnique({
        where: { id: decoded.sub },
        select: {
          id: true,
          name: true,
          email: true,
          role: true,
          // Don't select password for security
        },
      });

      console.log('Database query result:', !!user);

      if (!user) {
        console.log('User not found in database for ID:', decoded.sub);
        return null;
      }

      console.log('User verification successful');
      return user;
    } catch (error) {
      console.error('Token verification failed:', error.message);
      // Token is invalid or expired
      return null;
    }
  }
}
