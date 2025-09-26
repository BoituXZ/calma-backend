import {
  IsEnum,
  IsOptional,
  IsString,
  IsBoolean,
  IsInt,
  Min,
  Max,
} from 'class-validator';
import {
  AgeGroup,
  Location,
  EducationLevel,
  FamilyType,
  RespectLevel,
  EconomicLevel,
} from '@prisma/client';

export class CreateCulturalProfileDto {
  @IsEnum(AgeGroup)
  ageGroup: AgeGroup;

  @IsEnum(Location)
  location: Location;

  @IsEnum(EducationLevel)
  educationLevel: EducationLevel;

  @IsOptional()
  @IsString()
  ethnicBackground?: string;

  @IsOptional()
  @IsString()
  religiousBackground?: string;

  @IsOptional()
  @IsString()
  languagePreference?: string;

  @IsEnum(FamilyType)
  familyStructure: FamilyType;

  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(20)
  householdSize?: number;

  @IsBoolean()
  hasElders: boolean;

  @IsOptional()
  @IsString()
  communicationStyle?: string;

  @IsEnum(RespectLevel)
  respectLevel: RespectLevel;

  @IsEnum(EconomicLevel)
  economicStatus: EconomicLevel;

  @IsOptional()
  @IsString()
  employmentStatus?: string;
}

export class UpdateCulturalProfileDto {
  @IsOptional()
  @IsEnum(AgeGroup)
  ageGroup?: AgeGroup;

  @IsOptional()
  @IsEnum(Location)
  location?: Location;

  @IsOptional()
  @IsEnum(EducationLevel)
  educationLevel?: EducationLevel;

  @IsOptional()
  @IsString()
  ethnicBackground?: string;

  @IsOptional()
  @IsString()
  religiousBackground?: string;

  @IsOptional()
  @IsString()
  languagePreference?: string;

  @IsOptional()
  @IsEnum(FamilyType)
  familyStructure?: FamilyType;

  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(20)
  householdSize?: number;

  @IsOptional()
  @IsBoolean()
  hasElders?: boolean;

  @IsOptional()
  @IsString()
  communicationStyle?: string;

  @IsOptional()
  @IsEnum(RespectLevel)
  respectLevel?: RespectLevel;

  @IsOptional()
  @IsEnum(EconomicLevel)
  economicStatus?: EconomicLevel;

  @IsOptional()
  @IsString()
  employmentStatus?: string;
}
