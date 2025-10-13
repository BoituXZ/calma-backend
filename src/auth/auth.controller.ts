import {
  Controller,
  Post,
  Get,
  Body,
  Res,
  Req,
  HttpStatus,
  HttpCode,
} from '@nestjs/common';
import { AuthService } from './auth.service';
import { SignupDto } from 'src/common/dto/signup.dto';
import { LoginDto } from 'src/common/dto/login.dto';
import { Response, Request } from 'express';

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

    return res.status(HttpStatus.OK).json({
      message: 'Logged out successfully',
    });
  }

  // Add the new /me endpoint with debug logging
  @Get('me')
  async getCurrentUser(@Req() req: Request, @Res() res: Response) {
    try {
      // Debug: Log all cookies
      console.log('All cookies received:', req.cookies);
      console.log('Headers:', req.headers.cookie);

      // Extract JWT from cookie
      const token = req.cookies?.jwt;

      console.log('JWT token found:', !!token);
      if (token) {
        console.log('Token length:', token.length);
        console.log('Token starts with:', token.substring(0, 20) + '...');
      }

      if (!token) {
        console.log('No JWT token found in cookies');
        return res.status(HttpStatus.UNAUTHORIZED).json({
          message: 'No authentication token found',
        });
      }

      // Verify and get user info
      const user = await this.authService.verifyToken(token);

      console.log('User verification result:', !!user);
      if (user) {
        console.log('User found:', { id: user.id, email: user.email });
      }

      if (!user) {
        console.log('Token verification failed');
        return res.status(HttpStatus.UNAUTHORIZED).json({
          message: 'Invalid or expired token',
        });
      }

      console.log('Successfully returning user data');
      return res.status(HttpStatus.OK).json({ user });
    } catch (error) {
      console.error('Get current user error:', error);
      return res.status(HttpStatus.UNAUTHORIZED).json({
        message: 'Authentication failed',
      });
    }
  }
}
