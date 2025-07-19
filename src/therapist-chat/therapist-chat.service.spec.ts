import { Test, TestingModule } from '@nestjs/testing';
import { TherapistChatService } from './therapist-chat.service';

describe('TherapistChatService', () => {
  let service: TherapistChatService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [TherapistChatService],
    }).compile();

    service = module.get<TherapistChatService>(TherapistChatService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });
});
