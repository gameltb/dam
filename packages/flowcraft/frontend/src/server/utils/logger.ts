import os from "os";
import path from "path";
import winston from "winston";

const logFile = path.join(os.tmpdir(), "flowcraft-server.log");

const logger = winston.createLogger({
  defaultMeta: { service: "flowcraft-server" },
  format: winston.format.combine(
    winston.format.timestamp({ format: "YYYY-MM-DD HH:mm:ss" }),
    winston.format.errors({ stack: true }),
    winston.format.splat(),
    winston.format.json(),
  ),
  level: "info",
  transports: [
    new winston.transports.File({
      filename: logFile,
      maxFiles: 5,
      maxsize: 5242880, // 5MB
    }),
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.printf(({ level, message, timestamp, ...meta }) => {
          return `${timestamp} [${level}]: ${message} ${Object.keys(meta).length > 2 ? JSON.stringify(meta) : ""}`;
        }),
      ),
    }),
  ],
});

export default logger;
