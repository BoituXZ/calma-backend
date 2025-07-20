import { Module } from '@nestjs/common';
import { AuthService } from './auth.service';
import { AuthController } from './auth.controller';
import { DatabaseService } from 'src/database/database.service';
import { JwtModule } from '@nestjs/jwt'; 
import { jwtConstants } from 'src/constants/jwtConstants';


@Module({
  providers: [AuthService, DatabaseService],
  controllers: [AuthController],
  exports: [AuthService],
  imports: [
    JwtModule.register({
      secret: jwtConstants.secret,  // define your secret
      signOptions: { expiresIn: '30d' },  // or whatever you want
    }),
  ],
  
})
export class AuthModule {}
