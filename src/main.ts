import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import * as cookieParser from 'cookie-parser';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  
  // IMPORTANT: Add cookie parser FIRST
  app.use(cookieParser());
  
  // Define allowed origins
  const allowedOrigins = [
    'http://localhost:8080',  // Your frontend URL
    'http://localhost:5173',  // Alternative Vite port
    'http://localhost:3001',  // Common React port
  ];
app.setGlobalPrefix('api')
  // Enable CORS with cookie support
  app.enableCors({
    origin: allowedOrigins,
    credentials: true, // This is CRUCIAL for cookies
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS', 'PATCH'],
    allowedHeaders: [
      'Content-Type', 
      'Authorization', 
      'Accept', 
      'X-Requested-With',
      'Access-Control-Allow-Headers',
      'Origin',
      'Cookie',
    ],
    exposedHeaders: ['Set-Cookie'], 
    optionsSuccessStatus: 200,
  });

  const port = process.env.PORT || 3000;
  await app.listen(port);
  
  console.log(`Server running on http://localhost:${port}`);
  console.log(`CORS enabled for origins: ${allowedOrigins.join(', ')}`);
  console.log(`Cookie parser enabled`);
}

bootstrap();