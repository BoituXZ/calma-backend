import {
  Controller,
  Post,
  Body,
  Res,
  HttpStatus,
  HttpCode,
} from '@nestjs/common';
import { AuthService } from './auth.service';
import { SignupDto } from 'src/common/dto/signup.dto';
import { LoginDto } from 'src/common/dto/login.dto';
import { Response } from 'express';

@Controller('auth')
export class AuthController {
  constructor(private authService: AuthService) {}

  @Post('signup')
  async signup(@Body() signupDto: SignupDto, @Res() res: Response) {
    const result = await this.authService.signup(signupDto);
    res.cookie('jwt', result.access_token, {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
      maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
    });
    return res.status(HttpStatus.OK).json({
      message: result.message,
      user: result.user,
    });
  }

  @HttpCode(HttpStatus.OK)
  @Post('login')
  async login(@Body() loginDto: LoginDto, @Res() res: Response) {
    const result = await this.authService.login(loginDto);
    if ('access_token' in result) {
      res.cookie('jwt', result.access_token, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: 'strict',
        maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
      });
      return res.status(HttpStatus.OK).json({
        message: result.message,
        user: result.user,
      });
    }
    return res.status(HttpStatus.UNAUTHORIZED).json(result);
  }

  @HttpCode(HttpStatus.OK)
  @Post('logout')
  async logout(@Res() res: Response) {
    res.clearCookie('jwt', {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'strict',
    });
    return res
      .status(HttpStatus.OK)
      .json({ message: 'Logged out successfully' });
  }
}
