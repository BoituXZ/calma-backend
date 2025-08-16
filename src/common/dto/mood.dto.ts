import { IsNumber, IsString, isString } from "class-validator";


export class MoodDto {
    @IsString()
    id: string

    @IsNumber({}, {message: "Should be a number"})
    mood: number;
}