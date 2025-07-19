import { Test, TestingModule } from '@nestjs/testing';
import { SavedResourceController } from './saved-resource.controller';

describe('SavedResourceController', () => {
  let controller: SavedResourceController;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [SavedResourceController],
    }).compile();

    controller = module.get<SavedResourceController>(SavedResourceController);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });
});
