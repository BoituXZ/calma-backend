import {
  Body,
  Controller,
  Post,
  Res,
  Get,
  HttpStatus,
  HttpCode,
  Query,
} from '@nestjs/common';
import { MoodDto } from 'src/common/dto/mood.dto';
import { MoodService } from './mood.service';
import { Response } from 'express';

@Controller('mood')
export class MoodController {
  constructor(private moodService: MoodService) {}
  @Post('moods')
  async addMood(@Body() mood: MoodDto, @Res() res: Response) {
    try {
      const result = await this.moodService.addMood(mood);
      return res.status(HttpStatus.OK).json({
        message: 'Mood added',
      });
    } catch (error) {
      console.log('Error adding mood');
    }
  }

  @Get('moods')
  async fetchMoods(@Query('userId') userId: string, @Res() res: Response) {
    try {
      const result = await this.moodService.getMood(userId);
      return res.status(HttpStatus.OK).json({
        message: result,
      });
    } catch (error) {
      return res.status(HttpStatus.NOT_FOUND).json({
        message: 'No Moods available yet',
      });
    }
  }
}
// TODO: Finish off the post and get methods for moods.
