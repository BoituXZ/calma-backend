export interface CulturalProfile {
  id: string;
  ageGroup: string;
  location: string;
  educationLevel: string;
  ethnicBackground?: string;
  religiousBackground?: string;
  languagePreference?: string;
  familyStructure: string;
  householdSize?: number;
  hasElders: boolean;
  communicationStyle?: string;
  respectLevel: string;
  economicStatus: string;
  employmentStatus?: string;
}

export interface ConversationMemory {
  recentMessages: {
    id: string;
    message: string;
    sender: string;
    timestamp: Date;
    emotionalTone?: string;
    detectedTopics?: string[];
  }[];
  userTopics: {
    topic: string;
    frequency: number;
    severity: string;
    status: string;
    lastDiscussed: Date;
  }[];
  sessionContext: {
    primaryTopic?: string;
    emotionalState?: string;
    keyInsights?: any;
    actionItems: string[];
  };
  progressTracking: {
    relationshipStrength: number;
    trustLevel: number;
    improvementAreas: string[];
    concernAreas: string[];
  };
}

export interface AIServiceRequest {
  message: string;
  userId: string;
  sessionId?: string;
  culturalProfile: CulturalProfile;
  conversationMemory: ConversationMemory;
}

export interface AIServiceResponse {
  response: string;
  emotionalTone?: string;
  detectedTopics?: string[];
  culturalAdaptations?: any;
  memoryUpdates?: {
    keyInsights?: any;
    actionItems?: string[];
    topicUpdates?: {
      topic: string;
      severity?: string;
      status?: string;
    }[];
  };
  progressUpdates?: {
    relationshipStrength?: number;
    trustLevel?: number;
    improvementAreas?: string[];
    concernAreas?: string[];
  };
}
