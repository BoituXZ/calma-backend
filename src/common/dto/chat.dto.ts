import { IsString, IsOptional, IsUUID } from 'class-validator';

export class ChatRequestDto {
  @IsString()
  message: string;

  @IsUUID()
  userId: string;

  @IsOptional()
  @IsUUID()
  sessionId?: string;
}

export class ChatResponseDto {
  userMessage: {
    id: string;
    message: string;
    sender: string;
    timestamp: Date;
  };

  botMessage: {
    id: string;
    message: string;
    sender: string;
    timestamp: Date;
    emotionalTone?: string | null;
    detectedTopics?: string[];
  };

  session: {
    id: string;
    title?: string;
  };
}
