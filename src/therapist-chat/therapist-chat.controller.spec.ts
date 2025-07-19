import { Test, TestingModule } from '@nestjs/testing';
import { TherapistChatController } from './therapist-chat.controller';

describe('TherapistChatController', () => {
  let controller: TherapistChatController;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [TherapistChatController],
    }).compile();

    controller = module.get<TherapistChatController>(TherapistChatController);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });
});
