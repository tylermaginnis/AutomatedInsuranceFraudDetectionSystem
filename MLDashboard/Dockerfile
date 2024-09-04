# Use the official ASP.NET Core runtime as a parent image
FROM mcr.microsoft.com/dotnet/aspnet:7.0 AS base
WORKDIR /app
EXPOSE 80
EXPOSE 443

# Use the official ASP.NET Core SDK image to build the app
FROM mcr.microsoft.com/dotnet/sdk:7.0 AS build
WORKDIR /src
COPY ["MLDashboard/MLDashboard.csproj", "MLDashboard/"]
RUN dotnet restore "MLDashboard/MLDashboard.csproj"
COPY . .
WORKDIR "/src/MLDashboard"
RUN dotnet build "MLDashboard.csproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "MLDashboard.csproj" -c Release -o /app/publish

# Use the base image to run the app
FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "MLDashboard.dll"]
