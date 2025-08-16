import { Injectable, NotFoundException } from '@nestjs/common';
import { MoodDto } from 'src/common/dto/mood.dto';
import { DatabaseService } from 'src/database/database.service';

@Injectable()
export class MoodService {
  constructor(private databaseService: DatabaseService) {}

  async addMood(moodDto: MoodDto) {
    const { id, mood } = moodDto;
    try {
      // Converting the mood to an actual Value
      let moodText = '';
      switch (mood) {
        case 1:
          moodText = 'Very Low';
          break;
        case 2:
          moodText = 'Low';
          break;

        case 3:
          moodText = 'Neutral';
          break;

        case 4:
          moodText = 'Good';
          break;

        case 5:
          moodText = 'Very Good';
          break;
      }
      const addedMood = await this.databaseService.mood.create({
        data: {
          userId: id,
          mood: moodText,
        },
      });
    } catch (error) {
      console.log('Mood Creation Error', error);
      throw new NotFoundException();
    }
  }

  async getMood(userId: string) {
    const id = userId;

    try {
      const fetchedMoods = await this.databaseService.mood.findMany({
        where: {
          userId: id,
        },
        select: {
          mood: true,
        },
      });
      return fetchedMoods;
    } catch (error) {
      console.log('No moods', error);
      throw new NotFoundException();
    }
  }
}
