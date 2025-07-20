import {
  BadRequestException,
  Injectable,
  InternalServerErrorException,
  UnauthorizedException
} from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { PrismaClient, Prisma } from 'generated/prisma';
import { SignupDto } from 'src/common/dto/signup.dto';
import * as bcrypt from 'bcrypt';
import { LoginDto } from 'src/common/dto/login.dto';
import { DatabaseService } from 'src/database/database.service';

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

    const hash = await bcrypt.hash(dto.password, 10);

    try {
      const user = await this.database.user.create({
        data: {
          name: dto.name,
          email: dto.email,
          password: hash,
        },
        select: {
          id: true,
          name: true,
          email: true,
        },
      });

      const token = await this.jwtService.signAsync({
        sub: user.id,
        email: user.email,
      });

      return {
        message: 'User created successfully',
        access_token: token,
      };
    } catch (error) {
      // if (error instanceof this.database.databaseClientKnownRequestError) {
      //   if (error.code === 'P2002') {
      //     throw new BadRequestException('Email already exists');
      //   }
      // }
      throw new InternalServerErrorException('Could not create user');
    }
  }

  async login(dto: LoginDto) {
    const { email, password } = dto;

    try {
      const user = await this.database.user.findUnique({ where: { email } });

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
      };
    } catch (error) {
      console.error('Login error:', error);
      throw new InternalServerErrorException('Something went wrong');
    }
  }
}
