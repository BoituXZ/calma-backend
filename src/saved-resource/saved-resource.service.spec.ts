import { Test, TestingModule } from '@nestjs/testing';
import { SavedResourceService } from './saved-resource.service';

describe('SavedResourceService', () => {
  let service: SavedResourceService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [SavedResourceService],
    }).compile();

    service = module.get<SavedResourceService>(SavedResourceService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });
});
