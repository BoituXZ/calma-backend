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
    analysisResults?: {
      mood_detected?: string;
      confidence?: number;
      emotional_intensity?: number;
      suggested_resources?: string[];
      cultural_elements_detected?: string[];
      quality_metrics?: {
        word_count?: number;
        sentence_count?: number;
        cultural_awareness_score?: number;
        empathy_score?: number;
        response_length_category?: string;
      };
    };
  };

  session: {
    id: string;
    title?: string;
  };
}
