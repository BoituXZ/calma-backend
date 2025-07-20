import { IsEmail, MinLength } from "class-validator";


export class LoginDto {
    @IsEmail({}, {message:"Should be valid email"})
  email: string;

  @MinLength(8, { message: 'Password must be at least 8 characters long' })
  password: string;
}